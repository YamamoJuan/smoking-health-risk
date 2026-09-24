"""Konfigurasi terpusat untuk proyek Smoking Health Risk Prediction.

Seluruh path dibangun secara relatif terhadap root proyek sehingga kode
dapat dijalankan di mesin manapun tanpa absolute path.
"""
from pathlib import Path

# root proyek = folder 'smoking-health-risk'
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "framingham.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
ASSETS_DIR = PROJECT_ROOT / "assets"

TARGET = "TenYearCHD"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Nama fitur asli di dataset Framingham
NUMERIC_FEATURES = [
    "age",
    "cigsPerDay",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]

CATEGORICAL_FEATURES = [
    "male",           # 1 = laki-laki, 0 = perempuan
    "education",      # 1-4 (tingkat pendidikan)
    "BPMeds",         # sedang mengonsumsi obat tekanan darah
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "currentSmoker",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Label manusiawi untuk aplikasi Streamlit (Bahasa Indonesia)
FEATURE_LABELS = {
    "age": "Usia",
    "cigsPerDay": "Jumlah rokok per hari",
    "totChol": "Kolesterol total (mg/dL)",
    "sysBP": "Tekanan darah sistolik (mmHg)",
    "diaBP": "Tekanan darah diastolik (mmHg)",
    "BMI": "Indeks Massa Tubuh (BMI)",
    "heartRate": "Detak jantung (bpm)",
    "glucose": "Gula darah (mg/dL)",
    "male": "Jenis kelamin (laki-laki)",
    "education": "Tingkat pendidikan",
    "BPMeds": "Mengonsumsi obat tekanan darah",
    "prevalentStroke": "Riwayat stroke",
    "prevalentHyp": "Riwayat hipertensi",
    "diabetes": "Riwayat diabetes",
    "currentSmoker": "Perokok aktif saat ini",
}
