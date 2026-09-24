"""Preprocessing untuk Smoking Health Risk Prediction.

Modul ini menyediakan pipeline preprocessing sklearn yang identik untuk
training (src/train.py) dan inference (src/predict.py + app.py).
Pipeline dibangun lewat ColumnTransformer:
  * fitur numerik -> imputasi median + StandardScaler
  * fitur kategorikal -> imputasi most_frequent + OneHotEncoder
    (drop='if_binary' agar kolom biner menjadi satu kolom 0/1)
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Bangun ColumnTransformer preprocessing standar."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Kembalikan nama fitur hasil transformasi preprocessor (untuk SHAP/importance)."""
    names = []

    numeric_pipe = preprocessor.named_transformers_["num"]
    scaler = numeric_pipe.named_steps["scaler"]
    if hasattr(scaler, "get_feature_names_out"):
        names.extend(scaler.get_feature_names_out(NUMERIC_FEATURES))
    else:
        names.extend(NUMERIC_FEATURES)

    categorical_pipe = preprocessor.named_transformers_["cat"]
    onehot = categorical_pipe.named_steps["onehot"]
    names.extend(onehot.get_feature_names_out(CATEGORICAL_FEATURES))

    return list(names)
