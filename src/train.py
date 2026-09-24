"""Training pipeline untuk Smoking Health Risk Prediction.

Workflow:
    1. Load & validasi dataset
    2. Split train/test (stratified)
    3. Baseline model comparison (Logistic Regression, Random Forest,
       Gradient Boosting)
    4. Hyperparameter tuning untuk model kandidat via GridSearchCV
       (scoring=ROC-AUC, CV stratified)
    5. Evaluasi final di test set (Accuracy, Precision, Recall, F1, ROC-AUC)
    6. Feature importance + simpan model (joblib) + metrics JSON
"""
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from .config import (
    ALL_FEATURES,
    ASSETS_DIR,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURE_LABELS,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from .predict import load_model, predict
from .preprocessing import build_preprocessor, get_feature_names

warnings.filterwarnings("ignore", category=UserWarning)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


# ---------------------------------------------------------------------
# 1. Data loading & validation
# ---------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    required = ALL_FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib hilang: {missing}")
    return df


def basic_checks(df: pd.DataFrame) -> None:
    print(f"Dataset: {df.shape[0]} baris x {df.shape[1]} kolom")
    dup = df.duplicated().sum()
    print(f"Duplikat: {dup}")
    mv = df.isnull().sum()
    mv = mv[mv > 0]
    print("Missing values per kolom:")
    print(mv if not mv.empty else "  (tidak ada)")
    print(f"\nDistribusi target ({TARGET}):")
    print(df[TARGET].value_counts(normalize=False).sort_index())


# ---------------------------------------------------------------------
# 2. Model candidates & tuning grids
# ---------------------------------------------------------------------

def get_model_grids() -> dict:
    """Kembalikan dict: nama_model -> (estimator, param_grid)."""
    preprocessor = build_preprocessor()

    return {
        "Logistic Regression": (
            Pipeline([
                ("preprocess", preprocessor),
                ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            Pipeline([
                ("preprocess", build_preprocessor()),
                ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
            ]),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [4, 8, 12, None],
                "model__min_samples_split": [2, 5, 10],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Gradient Boosting": (
            Pipeline([
                ("preprocess", build_preprocessor()),
                ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
            },
        ),
    }


# ---------------------------------------------------------------------
# 3. Training & tuning
# ---------------------------------------------------------------------

def tune_models(X_train, y_train) -> dict:
    grids = get_model_grids()
    tuned = {}

    for name, (pipeline, param_grid) in grids.items():
        print(f"\n--- Tuning: {name} ---")
        gs = GridSearchCV(
            pipeline,
            param_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        tuned[name] = gs
        print(f"  Best CV ROC-AUC: {gs.best_score_:.4f}")
        print(f"  Best params: {gs.best_params_}")

    return tuned


# ---------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------

def evaluate_model(name, pipeline, X_test, y_test, threshold=0.5) -> dict:
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "threshold": threshold,
        "cv_roc_auc": None,  # diisi di main() jika tersedia
    }
    return metrics, y_pred, y_proba


def find_best_threshold(pipeline, X_train, y_train, cv=5) -> float:
    """Cari threshold yang memaksimalkan F1 pada out-of-fold CV.

    Threshold dipilih hanya dari data latih agar test set tidak bocor.
    """
    oof_proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.025):
        pred = (oof_proba >= t).astype(int)
        f1 = f1_score(y_train, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(round(best_t, 3))


def plot_confusion_matrix(cm, model_name, path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Risiko Rendah", "Risiko Tinggi"],
                yticklabels=["Risiko Rendah", "Risiko Tinggi"])
    plt.ylabel("Aktual")
    plt.xlabel("Prediksi")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def friendly_feature_name(name: str) -> str:
    """Konversi nama fitur transformed menjadi label manusiawi untuk plot."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name in NUMERIC_FEATURES:
        return FEATURE_LABELS.get(name, name)
    if "_" in name:
        base = name.rsplit("_", 1)[0]
        if base in CATEGORICAL_FEATURES:
            return FEATURE_LABELS.get(base, base)
    return name


def plot_feature_importance(feature_names, importances, model_name, path, top_n=15):
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    fi.index = [friendly_feature_name(n) for n in fi.index]
    plt.figure(figsize=(7, 5))
    sns.barplot(x=fi.values, y=fi.index, hue=fi.index, palette="viridis", legend=False)
    plt.title(f"Feature Importance — {model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Fitur")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_comparison(comparison_df: pd.DataFrame, path):
    ax = comparison_df[["roc_auc", "f1", "recall", "precision", "accuracy"]].plot(
        kind="bar", figsize=(9, 5), rot=0
    )
    ax.set_title("Perbandingan Model (Test Set)")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ---------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------

def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_data()
    basic_checks(df)

    # Drop baris dengan target NaN (jika ada)
    df = df.dropna(subset=[TARGET])

    X = df[ALL_FEATURES]
    y = df[TARGET]

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # 3. Tuning semua kandidat (CV pada train set)
    tuned = tune_models(X_train, y_train)

    # 4. Evaluasi di test set dengan threshold optimal (dipilih dari train CV)
    comparison = {}
    predictions = {}
    thresholds = {}
    for name, gs in tuned.items():
        pipeline = gs.best_estimator_
        threshold = find_best_threshold(pipeline, X_train, y_train)
        thresholds[name] = threshold
        metrics, y_pred, y_proba = evaluate_model(name, pipeline, X_test, y_test, threshold=threshold)
        metrics["cv_roc_auc"] = gs.best_score_
        comparison[name] = metrics
        predictions[name] = (y_pred, y_proba, pipeline)

    print("\nThreshold optimal (dari train CV):")
    for name, t in thresholds.items():
        print(f"  {name}: {t}")

    comp_df = pd.DataFrame(comparison).T
    print("\n===== Perbandingan Model (Test Set) =====")
    print(comp_df.round(4).to_string())

    # 5. Pilih model terbaik berdasarkan ROC-AUC test
    #    (ROC-AUC threshold-independent, lalu threshold optimal dipakai untuk klasifikasi)
    #    Filter dulu model dengan ROC-AUC test >= 0.5 agar tidak memilih model buruk
    valid_models = comp_df[comp_df["roc_auc"] >= 0.5]
    if valid_models.empty:
        raise RuntimeError("Semua model memiliki ROC-AUC < 0.5; periksa pipeline.")
    best_name = valid_models["roc_auc"].idxmax()
    best_metrics = comparison[best_name]
    y_pred_best, y_proba_best, best_pipeline = predictions[best_name]

    print(f"\nModel terbaik (berdasarkan ROC-AUC test): {best_name}")
    print(f"  ROC-AUC : {best_metrics['roc_auc']:.4f}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall  : {best_metrics['recall']:.4f}")
    print(f"  F1      : {best_metrics['f1']:.4f}")

    # 6. Confusion matrix
    y_pred_best = (y_proba_best >= thresholds[best_name]).astype(int)
    best_metrics, _, _ = evaluate_model(best_name, best_pipeline, X_test, y_test,
                                        threshold=thresholds[best_name])
    comparison[best_name] = best_metrics
    comp_df = pd.DataFrame(comparison).T
    comparison[best_name]["cv_roc_auc"] = tuned[best_name].best_score_
    cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1])
    plot_confusion_matrix(cm, best_name, ASSETS_DIR / "confusion_matrix.png")

    # 7. Feature importance
    preprocessor = best_pipeline.named_steps["preprocess"]
    feature_names = get_feature_names(preprocessor)
    model = best_pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = None

    fi_df = None
    if importances is not None:
        plot_feature_importance(feature_names, importances, best_name,
                                ASSETS_DIR / "feature_importance.png")
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})

    # 8. Plot perbandingan model
    plot_comparison(comp_df, ASSETS_DIR / "model_comparison.png")

    # 9. Simpan model + metrics
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nModel disimpan di: {MODEL_PATH}")

    metrics_out = {
        "dataset": "Framingham Heart Study (n=4240, 15 fitur + target)",
        "target": "TenYearCHD (1 = risiko tinggi CHD 10 tahun)",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "best_model": best_name,
        "optimal_threshold": thresholds[best_name],
        "comparison": comparison,
        "class_distribution": y.value_counts().to_dict(),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False, default=str)
    print(f"Metrics disimpan di: {METRICS_PATH}")

    # 10. Sanity check: reload + predict sample
    reloaded = load_model()
    sample = X_test.head(5)
    assert np.array_equal(reloaded.predict(sample), best_pipeline.predict(sample)), \
        "Reload model menghasilkan prediksi berbeda!"
    print("Reload check: OK")

    # 11. Contoh prediksi sample
    res = predict(reloaded, X_test.head(3))
    print("\nContoh prediksi 3 sample:")
    print(res[["pred_class", "risk_score", "risiko"]].to_string(index=False))


if __name__ == "__main__":
    main()
