#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PROJECT 3 OF 3: SUPERVISED MACHINE LEARNING - CLASSIFICATION
================================================================================
Kurikulum : Course 3 - Supervised Machine Learning: Classification
Studi Kasus: Klasifikasi Status Kemiskinan & Klaster IPM Jawa Tengah (BPS 2025)
Cakupan Materi Course 3:
  Perumusan Target Kategorikal & Stratified Train-Test Split
  Logistic Regression (Probabilitas Sigmoid, Log-Odds & Regularisasi L2)
  K-Nearest Neighbors (KNN) Classifier & Pencarian Nilai k Terbaik
  Support Vector Machine (Linear SVM Hinge-Loss Classifier)
  Decision Tree & Random Forest Classifier (Gini Impurity Ensemble)
  Evaluasi Komprehensif: Confusion Matrix (TP, TN, FP, FN), Accuracy,
    Precision, Recall (Sensitivity), Specificity, F1-Score, & ROC-AUC
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


class LogisticRegressionClassifier:
    """Logistic Regression dengan regularisasi L2 dan fungsi aktivasi Sigmoid."""
    def __init__(self, C=1.0, lr=0.1, max_iter=1500):
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = 0.0

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        w = np.zeros(p, dtype=float)
        b = 0.0
        reg = 1.0 / max(self.C, 1e-6)
        for _ in range(self.max_iter):
            probs = self._sigmoid(X @ w + b)
            err = probs - y
            w -= self.lr * ((X.T @ err) / n + (reg / n) * w)
            b -= self.lr * float(np.mean(err))
        self.coef_ = w
        self.intercept_ = b
        return self

    def predict_proba(self, X):
        p1 = self._sigmoid(np.asarray(X, dtype=float) @ self.coef_ + self.intercept_)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class KNNClassifier:
    """K-Nearest Neighbors Classifier dengan pembobotan jarak Euclidean."""
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_tr = None
        self.y_tr = None

    def fit(self, X, y):
        self.X_tr = np.asarray(X, dtype=float)
        self.y_tr = np.asarray(y, dtype=int)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        probs = []
        for row in X:
            dists = np.linalg.norm(self.X_tr - row, axis=1)
            idx = np.argsort(dists)[: self.n_neighbors]
            w = 1.0 / (dists[idx] + 1e-6)
            p1 = float(np.sum(w * (self.y_tr[idx] == 1)) / np.sum(w))
            probs.append([1.0 - p1, p1])
        return np.array(probs)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class LinearSVMClassifier:
    """Support Vector Machine (SVM) Linear menggunakan Sub-Gradient Descent Hinge Loss."""
    def __init__(self, C=1.0, lr=0.01, max_iter=2000):
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y_signed = np.where(np.asarray(y, dtype=int) == 1, 1.0, -1.0)
        n, p = X.shape
        w = np.zeros(p, dtype=float)
        b = 0.0
        for _ in range(self.max_iter):
            margins = y_signed * (X @ w + b)
            misclassified = margins < 1.0
            dw = w / n - self.C * (X[misclassified].T @ y_signed[misclassified]) / n
            db = -self.C * float(np.sum(y_signed[misclassified])) / n
            w -= self.lr * dw
            b -= self.lr * db
        self.coef_ = w
        self.intercept_ = b
        return self

    def predict_proba(self, X):
        scores = np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
        p1 = 1.0 / (1.0 + np.exp(-np.clip(scores, -250, 250)))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        scores = np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
        return (scores >= 0.0).astype(int)


class DecisionTreeGiniClassifier:
    """Decision Tree Classifier berbasis Gini Impurity."""
    def __init__(self, max_depth=4, min_samples_split=3, max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng or np.random.default_rng(42)
        self.root = None
        self.feature_importances_ = None

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        p = np.mean(y == 1)
        return 2.0 * p * (1.0 - p)

    def _build(self, X, y, depth):
        n, p = X.shape
        if depth >= self.max_depth or n < self.min_samples_split or len(np.unique(y)) == 1:
            return {"leaf": True, "prob": float(np.mean(y == 1))}

        parent_g = self._gini(y)
        best_gain, best_f, best_t = 0.0, None, None
        k_feat = self.max_features or p
        feat_idx = self.rng.choice(p, size=min(p, k_feat), replace=False)

        for f in feat_idx:
            vals = np.unique(X[:, f])
            if len(vals) <= 1:
                continue
            for t in (vals[:-1] + vals[1:]) / 2.0:
                left = X[:, f] <= t
                right = ~left
                if np.sum(left) == 0 or np.sum(right) == 0:
                    continue
                gain = parent_g - (np.sum(left) / n) * self._gini(y[left]) - (np.sum(right) / n) * self._gini(y[right])
                if gain > best_gain:
                    best_gain, best_f, best_t = gain, int(f), float(t)

        if best_f is None:
            return {"leaf": True, "prob": float(np.mean(y == 1))}

        self.feature_importances_[best_f] += best_gain * n
        left_m = X[:, best_f] <= best_t
        return {
            "leaf": False,
            "f": best_f,
            "t": best_t,
            "l": self._build(X[left_m], y[left_m], depth + 1),
            "r": self._build(X[~left_m], y[~left_m], depth + 1),
        }

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        self.root = self._build(X, y, 0)
        tot = np.sum(self.feature_importances_)
        if tot > 0:
            self.feature_importances_ /= tot
        return self

    def _pred_one(self, r, node):
        if node["leaf"]:
            return node["prob"]
        return self._pred_one(r, node["l"] if r[node["f"]] <= node["t"] else node["r"])

    def predict_proba(self, X):
        p1 = np.array([self._pred_one(r, self.root) for r in np.asarray(X, dtype=float)])
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class RandomForestGiniClassifier:
    """Random Forest Classifier berbasis Bootstrap Aggregating (Bagging)."""
    def __init__(self, n_estimators=150, max_depth=4, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees_ = []
        self.feature_importances_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        imp = np.zeros(X.shape[1], dtype=float)
        n, p = X.shape
        max_f = max(1, int(np.sqrt(p)))
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            tree = DecisionTreeGiniClassifier(max_depth=self.max_depth, max_features=max_f, rng=rng)
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
            imp += tree.feature_importances_
        self.feature_importances_ = imp / np.sum(imp) if np.sum(imp) > 0 else imp
        return self

    def predict_proba(self, X):
        p1 = np.mean([t.predict_proba(X)[:, 1] for t in self.trees_], axis=0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def calc_roc_auc(y_true, y_Score):
    """Menghitung Area Under the ROC Curve (ROC-AUC) dengan Mann-Whitney U statistic."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_Score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    count = 0.0
    for p in pos:
        count += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(count / (len(pos) * len(neg)))


def eval_clf(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    auc = calc_roc_auc(y_true, y_prob)
    return acc, prec, rec, spec, f1, auc, [[tn, fp], [fn, tp]]


def loocv_clf(factory, X_df, y_ser, scale=True):
    X = X_df.values
    y = y_ser.values
    n = len(y)
    preds = np.zeros(n, dtype=int)
    probs = np.zeros(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr, y_tr = X[mask], y[mask]
        X_te = X[~mask]
        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
        m = factory().fit(X_tr, y_tr)
        preds[i] = int(m.predict(X_te)[0])
        probs[i] = float(m.predict_proba(X_te)[0, 1])
    return preds, probs


def run_classification_project():
    print("=" * 90)
    print("PROJECT 3: SUPERVISED MACHINE LEARNING - CLASSIFICATION")
    print("Studi Kasus: Klasifikasi Status Kemiskinan & Klaster Pembangunan IPM Jawa Tengah 2025")
    print("=" * 90)

    df = pd.read_csv(locate_dataset())
    df["hls_rata_rata"] = (df["hls_laki_laki_2025"] + df["hls_perempuan_2025"]) / 2.0
    df["uhh_rata_rata"] = (df["uhh_laki_laki_2025"] + df["uhh_perempuan_2025"]) / 2.0
    df["gap_hls_gender"] = df["hls_perempuan_2025"] - df["hls_laki_laki_2025"]
    df["log_pdrb"] = np.log(df["pdrb_adhb_2025_triliun_rp"])
    df["is_kota"] = (df["tipe_wilayah"] == "Kota").astype(int)

    # Target 1: Status Kemiskinan Wilayah (> 9,48% Rata-rata Provinsi Jawa Tengah)
    df["status_kemiskinan_tinggi"] = (df["persentase_penduduk_miskin_2025"] > 9.48).astype(int)
    # Target 2: Klaster IPM Maju (>= 75,0) vs Berkembang (< 75,0)
    df["klaster_ipm_maju"] = (df["ipm_2025"] >= 75.0).astype(int)

    # --------------------------------------------------------------------------
    # SKENARIO 1: KLASIFIKASI STATUS KEMISKINAN WILAYAH (> 9,48% vs <= 9,48%)
    # --------------------------------------------------------------------------
    features_pov = ["ipm_2025", "hls_rata_rata", "uhh_rata_rata", "log_pdrb", "gap_hls_gender", "is_kota"]
    X_pov = df[features_pov]
    y_pov = df["status_kemiskinan_tinggi"]

    models = [
        ("1. Logistic Regression (L2)", lambda: LogisticRegressionClassifier(C=1.0), True),
        ("2. K-Nearest Neighbors (k=5)", lambda: KNNClassifier(n_neighbors=5), True),
        ("3. Support Vector Machine (SVM)", lambda: LinearSVMClassifier(C=1.0), True),
        ("4. Decision Tree (Gini)", lambda: DecisionTreeGiniClassifier(max_depth=3), False),
        ("5. Random Forest Classifier", lambda: RandomForestGiniClassifier(n_estimators=150, max_depth=4), False),
    ]

    print("\n[STEP 1] Evaluasi 5 Algoritma Klasifikasi pada Skenario 1: Status Kemiskinan (> 9,48%)")
    print("         Distribusi Kelas: 19 Wilayah Kemiskinan Rendah (<=9,48%) vs 16 Wilayah Kemiskinan Tinggi (>9,48%)")
    print("-" * 90)
    print(f"{'Algoritma Klasifikasi':<33} | {'LOOCV Acc':>9} | {'Precision':>9} | {'Recall':>8} | {'Specificity':>11} | {'F1-Score':>8} | {'ROC-AUC':>7}")
    print("-" * 90)

    best_cm = None
    for name, factory, use_scale in models:
        preds, probs = loocv_clf(factory, X_pov, y_pov, scale=use_scale)
        acc, prec, rec, spec, f1, auc, cm = eval_clf(y_pov.values, preds, probs)
        if "Random Forest" in name:
            best_cm = cm
        print(f"{name:<33} | {acc*100:>8.2f}% | {prec*100:>8.2f}% | {rec*100:>7.2f}% | {spec*100:>10.2f}% | {f1*100:>7.2f}% | {auc:>7.4f}")

    print("\n         Detail Confusion Matrix (Random Forest Classifier - LOOCV 35 Kabupaten/Kota):")
    print(f"         - True Negative  (Aktual Rendah, Prediksi Rendah) : {best_cm[0][0]} Wilayah")
    print(f"         - False Positive (Aktual Rendah, Prediksi Tinggi) : {best_cm[0][1]} Wilayah")
    print(f"         - False Negative (Aktual Tinggi, Prediksi Rendah) : {best_cm[1][0]} Wilayah")
    print(f"         - True Positive  (Aktual Tinggi, Prediksi Tinggi) : {best_cm[1][1]} Wilayah")

    # --------------------------------------------------------------------------
    # STEP 2: HYPERPARAMETER TUNING NILAI K PADA K-NEAREST NEIGHBORS (KNN)
    # --------------------------------------------------------------------------
    print("\n[STEP 2] Hyperparameter Tuning Jumlah Tetangga Terdekat (k) pada Algoritma KNN")
    for k in [3, 5, 7, 9]:
        preds_k, probs_k = loocv_clf(lambda: KNNClassifier(n_neighbors=k), X_pov, y_pov, scale=True)
        acc_k, _, _, _, f1_k, auc_k, _ = eval_clf(y_pov.values, preds_k, probs_k)
        print(f"         * KNN (k = {k}) -> LOOCV Accuracy: {acc_k*100:6.2f}% | F1-Score: {f1_k*100:6.2f}% | ROC-AUC: {auc_k:.4f}")

    # --------------------------------------------------------------------------
    # SKENARIO 2: KLASIFIKASI KLASTER PEMBANGUNAN WILAYAH (IPM >= 75,0 vs < 75,0)
    # --------------------------------------------------------------------------
    features_ipm = ["hls_rata_rata", "uhh_rata_rata", "persentase_penduduk_miskin_2025", "log_pdrb", "is_kota"]
    X_ipm = df[features_ipm]
    y_ipm = df["klaster_ipm_maju"]

    print("\n[STEP 3] Evaluasi 5 Algoritma Klasifikasi pada Skenario 2: Klaster IPM Maju (>= 75,0)")
    print("         Distribusi Kelas: 18 Wilayah IPM Berkembang (<75,0) vs 17 Wilayah IPM Maju (>=75,0)")
    print("-" * 90)
    print(f"{'Algoritma Klasifikasi':<33} | {'LOOCV Acc':>9} | {'Precision':>9} | {'Recall':>8} | {'Specificity':>11} | {'F1-Score':>8} | {'ROC-AUC':>7}")
    print("-" * 90)

    for name, factory, use_scale in models:
        preds, probs = loocv_clf(factory, X_ipm, y_ipm, scale=use_scale)
        acc, prec, rec, spec, f1, auc, _ = eval_clf(y_ipm.values, preds, probs)
        print(f"{name:<33} | {acc*100:>8.2f}% | {prec*100:>8.2f}% | {rec*100:>7.2f}% | {spec*100:>10.2f}% | {f1*100:>7.2f}% | {auc:>7.4f}")

    rf_full = RandomForestGiniClassifier(n_estimators=150, max_depth=4).fit(X_ipm.values, y_ipm.values)
    print("\n         Feature Importance Random Forest (Faktor Penentu Klaster IPM Maju >= 75,0):")
    for feat, imp in sorted(zip(features_ipm, rf_full.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"         * {feat:<35}: {imp*100:5.2f}%")

    print("\n" + "=" * 90)
    print("PROJECT 3 (SUPERVISED MACHINE LEARNING: CLASSIFICATION) SELESAI DENGAN SUKSES!")
    print("=" * 90)


if __name__ == "__main__":
    run_classification_project()