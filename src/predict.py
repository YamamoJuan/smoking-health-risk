"""Inference helper untuk Smoking Health Risk Prediction."""
import json

import numpy as np
import pandas as pd

from .config import ALL_FEATURES, METRICS_PATH, MODEL_PATH, TARGET

# Label kelas target (0 = risiko rendah, 1 = risiko tinggi).
CLASS_NAMES = ["Risiko Rendah", "Risiko Tinggi"]

DEFAULT_THRESHOLD = 0.5


def load_threshold() -> float:
    """Baca threshold optimal dari models/metrics.json.

    Threshold dipilih saat training menggunakan out-of-fold CV pada train set
    sehingga tidak menyebabkan kebocoran test set.
    """
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        try:
            return float(metrics["optimal_threshold"])
        except (KeyError, TypeError, ValueError):
            pass
    return DEFAULT_THRESHOLD


def load_model():
    """Muat pipeline (preprocessing + model) dari models/model.joblib."""
    import joblib

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan di {MODEL_PATH}. "
            "Jalankan `python -m src.train` terlebih dahulu."
        )
    return joblib.load(MODEL_PATH)


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validasi DataFrame input agar kolomnya sesuai kontrak model."""
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom input tidak lengkap: {missing}. "
            f"Kolom yang diperlukan: {ALL_FEATURES}"
        )
    return df[ALL_FEATURES].copy()


def predict(model, df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    """Prediksi risiko untuk satu/beberapa baris data mentah.

    Mengembalikan DataFrame berisi: pred_class, risk_score, risiko.
    Threshold default diambil dari metrics.json; gunakan 0.5 jika metrics
    tidak tersedia.
    """
    X = validate_input(df)

    if threshold is None:
        threshold = load_threshold()

    proba = model.predict_proba(X)

    classes = list(model.classes_)
    pos_idx = classes.index(1) if 1 in classes else 0
    risk_score = proba[:, pos_idx]
    pred_class = (risk_score >= threshold).astype(int)

    results = pd.DataFrame({
        "pred_class": pred_class,
        "risk_score": risk_score,
    })
    results["risiko"] = results["pred_class"].map(
        {0: CLASS_NAMES[0], 1: CLASS_NAMES[1]}
    )
    return results


def predict_single(model, data: dict, threshold: float = None) -> dict:
    """Prediksi satu sample dari dict input (dipakai app.py)."""
    df = pd.DataFrame([data])
    result = predict(model, df, threshold=threshold)
    return result.iloc[0].to_dict()
