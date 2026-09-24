"""Aplikasi Streamlit: Smoking Health Risk Prediction.

Jalankan dengan:
    streamlit run app.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    ALL_FEATURES,
    ASSETS_DIR,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURE_LABELS,
    METRICS_PATH,
    NUMERIC_FEATURES,
    TARGET,
)
from src.predict import load_model, load_threshold, predict_single
from src.preprocessing import get_feature_names

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Prediksi Risiko Kesehatan Akibat Merokok",
    page_icon="🚭",
    layout="wide",
)


# ---------------------------------------------------------------------
# Load artifacts (cache)
# ---------------------------------------------------------------------

@st.cache_resource
def load_pipeline():
    return load_model()


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def friendly_feature_name(name: str) -> str:
    """Konversi nama fitur transformed (mis. 'male_1.0') menjadi label manusiawi."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name in NUMERIC_FEATURES:
        return FEATURE_LABELS.get(name, name)
    # fitur kategorikal yang sudah di-one-hot, mis. 'male_1.0' -> 'male'
    if "_" in name:
        base = name.rsplit("_", 1)[0]
        if base in CATEGORICAL_FEATURES:
            return FEATURE_LABELS.get(base, base)
    return name


def get_feature_importance(pipeline):
    try:
        preprocessor = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        feature_names = get_feature_names(preprocessor)
        if hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
        elif hasattr(model, "coef_"):
            vals = np.abs(model.coef_[0])
        else:
            return None
        return pd.DataFrame({"fitur": feature_names, "importance": vals}).sort_values(
            "importance", ascending=False
        )
    except Exception:
        return None


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------

st.sidebar.title("Navigasi")
page = st.sidebar.radio(
    "Pilih halaman:",
    ["🏠 Beranda", "🔮 Prediksi Risiko", "📊 Analisis Data", "ℹ️ Tentang Model"],
)

# ---------------------------------------------------------------------
# 1. BERANDA
# ---------------------------------------------------------------------

if page == "🏠 Beranda":
    st.title("Prediksi Risiko Kesehatan Akibat Kebiasaan Merokok")
    st.subheader("Estimasi risiko penyakit jantung koroner (CHD) 10 tahun berdasarkan data kesehatan & kebiasaan merokok")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Tujuan Project

        Project ini bertujuan memprediksi **risiko seseorang mengalami
        penyakit jantung koroner (CHD) dalam 10 tahun ke depan** berdasarkan
        faktor-faktor kesehatan seperti usia, kebiasaan merokok, tekanan
        darah, kolesterol, dan lainnya.

        Model yang digunakan adalah **machine learning klasifikasi biner** yang
        mempelajari pola dari data historis **Framingham Heart Study**
        (4.240 partisipan). Hasil model ditampilkan dalam bentuk **probabilitas
        risiko**, bukan diagnosis medis.

        ### Cara Kerja Model

        1. Data pasien diproses melalui pipeline preprocessing
           (imputasi missing value + scaling + one-hot encoding).
        2. Model mengeluarkan probabilitas risiko CHD 10 tahun.
        3. Probabilitas dikonversi menjadi kategori risiko:
           - **Risiko Rendah** (0)
           - **Risiko Tinggi** (1)

        ### Dataset

        Dataset yang digunakan adalah **Framingham Heart Study**,
        salah satu studi kohort epidemiologi terlama di dunia (dimulai 1948).
        Data ini mencakup 15 variabel klinis, perilaku, dan demografis untuk
        memprediksi kejadian CHD dalam 10 tahun.

        - Jumlah data: 4.240 baris
        - Target: `TenYearCHD` (1 = mengalami CHD dalam 10 tahun)
        - Ketidakseimbangan kelas: ~85% risiko rendah, ~15% risiko tinggi

        ### ⚠️ Disclaimer

        Hasil prediksi model ini **bukan diagnosis medis** dan tidak dapat
        menggantikan konsultasi dengan tenaga medis profesional. Model hanya
        memberikan estimasi statistik berdasarkan data historis dan tidak
        mempertimbangkan faktor individu yang tidak tercakup dalam dataset.
        """)

    with col2:
        st.info("""
        **Catatan Penting**

        - Model dilatih dengan metrik ROC-AUC sebagai acuan utama
          karena dataset bersifat *imbalanced*.
        - Semua angka evaluasi berasal dari eksperimen aktual, bukan asumsi.
        - Simpan model (`joblib`) agar pipeline preprocessing konsisten
          antara training dan inference.
        """)

        # Ringkasan metrik
        metrics = load_metrics()
        if metrics:
            best = metrics["best_model"]
            comp = metrics["comparison"][best]
        st.metric("Model Terbaik", best)
        st.metric("ROC-AUC (test)", f"{comp['roc_auc']:.3f}")
        st.metric("Recall", f"{comp['recall']:.3f}")

# ---------------------------------------------------------------------
# 2. PREDIKSI RISIKO
# ---------------------------------------------------------------------

elif page == "🔮 Prediksi Risiko":
    st.title("Prediksi Risiko Kesehatan")
    st.markdown(
        "Masukkan data kesehatan Anda di bawah ini. "
        "Model akan menampilkan estimasi probabilitas risiko penyakit jantung "
        "koroner (CHD) dalam 10 tahun ke depan."
    )

    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Usia (tahun)", 20, 80, 40)
            male = st.radio("Jenis Kelamin", ["Perempuan", "Laki-laki"], index=0)
            education = st.selectbox(
                "Tingkat Pendidikan",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1 - Tidak sekolah / SD",
                    2: "2 - SMP / SMA",
                    3: "3 - Diploma / Sarjana",
                    4: "4 - Pascasarjana",
                }[x],
            )
            currentSmoker = st.radio("Perokok aktif saat ini?", ["Tidak", "Ya"])

        with col2:
            cigsPerDay = st.slider(
                "Jumlah rokok per hari", 0, 70, 0,
                help="0 jika tidak merokok"
            )
            BPMeds = st.radio("Sedang mengonsumsi obat tekanan darah?", ["Tidak", "Ya"])
            prevalentStroke = st.radio("Riwayat stroke?", ["Tidak", "Ya"])
            prevalentHyp = st.radio("Riwayat hipertensi?", ["Tidak", "Ya"])
            diabetes = st.radio("Riwayat diabetes?", ["Tidak", "Ya"])

        with col3:
            totChol = st.number_input("Kolesterol total (mg/dL)", 100, 700, 200)
            sysBP = st.number_input("Tekanan darah sistolik (mmHg)", 80, 300, 120)
            diaBP = st.number_input("Tekanan darah diastolik (mmHg)", 40, 200, 80)
            BMI = st.number_input("Indeks Massa Tubuh (BMI)", 10.0, 60.0, 24.0, step=0.1)
            heartRate = st.number_input("Detak jantung (bpm)", 40, 200, 72)
            glucose = st.number_input("Gula darah (mg/dL)", 40, 500, 80)

        submitted = st.form_submit_button("🔮 Prediksi Risiko")

    if submitted:
        try:
            model = load_pipeline()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        input_data = {
            "age": age,
            "male": 1 if male == "Laki-laki" else 0,
            "education": education,
            "currentSmoker": 1 if currentSmoker == "Ya" else 0,
            "cigsPerDay": cigsPerDay,
            "BPMeds": 1 if BPMeds == "Ya" else 0,
            "prevalentStroke": 1 if prevalentStroke == "Ya" else 0,
            "prevalentHyp": 1 if prevalentHyp == "Ya" else 0,
            "diabetes": 1 if diabetes == "Ya" else 0,
            "totChol": totChol,
            "sysBP": sysBP,
            "diaBP": diaBP,
            "BMI": BMI,
            "heartRate": heartRate,
            "glucose": glucose,
        }

        result = predict_single(model, input_data)
        risk_score = result["risk_score"]
        risk_label = result["risiko"]
        threshold = load_threshold()

        st.markdown("---")
        st.subheader("Hasil Prediksi")

        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Kategori Risiko", risk_label)
        col_res2.metric("Probabilitas Risiko CHD", f"{risk_score:.1%}")
        col_res3.metric("Probabilitas Aman", f"{(1 - risk_score):.1%}")
        st.caption(f"Threshold klasifikasi yang digunakan: {threshold:.2f}")

        # Progress bar probabilitas
        st.progress(float(risk_score))

        if risk_score >= threshold:
            st.warning(
                "⚠️ Hasil model menunjukkan probabilitas risiko yang relatif tinggi. "
                "Ini adalah estimasi statistik, **bukan diagnosis medis**. "
                "Konsultasikan dengan dokter untuk pemeriksaan lebih lanjut."
            )
        else:
            st.success(
                "✅ Hasil model menunjukkan probabilitas risiko yang relatif rendah. "
                "Namun tetap jaga kesehatan dan periksakan diri secara berkala."
            )

        # Penjelasan sederhana berdasarkan feature importance
        st.markdown("### Faktor yang Berkontribusi Terhadap Prediksi")
        fi_df = get_feature_importance(model)
        if fi_df is not None:
            top_fi = fi_df.head(8).copy()
            top_fi["fitur"] = top_fi["fitur"].map(friendly_feature_name)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(
                top_fi["fitur"],
                top_fi["importance"],
                color=plt.cm.viridis(np.linspace(0, 1, len(top_fi))),
            )
            ax.invert_yaxis()
            ax.set_xlabel("Kontribusi (importance)")
            ax.set_title("Faktor Paling Berpengaruh (Model Global)")
            st.pyplot(fig)
            st.caption(
                "Catatan: feature importance menunjukkan fitur yang paling "
                "berkontribusi terhadap prediksi model secara umum, bukan "
                "penyebab langsung hasil pada individu tertentu."
            )
        else:
            st.info("Feature importance tidak tersedia untuk model ini.")

# ---------------------------------------------------------------------
# 3. ANALISIS DATA
# ---------------------------------------------------------------------

elif page == "📊 Analisis Data":
    st.title("Analisis Data")
    df = load_dataset()
    if df is None:
        st.error("Dataset tidak ditemukan.")
        st.stop()

    st.subheader("Cuplikan Dataset")
    st.dataframe(df.head(100), use_container_width=True)

    # Distribusi target
    st.subheader("Distribusi Target (TenYearCHD)")
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET].value_counts().sort_index()
    ax.bar(["Risiko Rendah (0)", "Risiko Tinggi (1)"], counts.values, color=["#4CAF50", "#F44336"])
    ax.set_ylabel("Jumlah")
    ax.set_title("Distribusi Kelas Target")
    st.pyplot(fig)

    # Distribusi usia
    st.subheader("Distribusi Usia")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["age"].dropna(), bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Usia")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)

    # Rokok per hari
    st.subheader("Distribusi Jumlah Rokok per Hari")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["cigsPerDay"].dropna(), bins=30, edgecolor="black", alpha=0.7, color="orange")
    ax.set_xlabel("Rokok per hari")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)

    # Hubungan smoking dengan target
    st.subheader("Kebiasaan Merokok vs Risiko CHD")
    smoke_risk = df.groupby("currentSmoker")[TARGET].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Tidak Merokok", "Merokok"], smoke_risk.values, color=["#4CAF50", "#FF9800"])
    ax.set_ylabel("Proporsi Risiko Tinggi")
    ax.set_title("Proporsi Risiko CHD 10 Tahun Berdasarkan Status Merokok")
    st.pyplot(fig)

    # Metrik model
    st.subheader("Performa Model")
    metrics = load_metrics()
    if metrics:
        comp = metrics["comparison"]
        comp_df = pd.DataFrame(comp).T
        st.dataframe(comp_df.round(4), use_container_width=True)

        st.markdown(f"**Model terbaik:** {metrics['best_model']}")

        # Confusion matrix
        cm_path = ASSETS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.subheader("Confusion Matrix")
            st.image(str(cm_path), caption=f"Confusion Matrix — {metrics['best_model']}", width=500)

        # Model comparison
        comp_path = ASSETS_DIR / "model_comparison.png"
        if comp_path.exists():
            st.subheader("Perbandingan Model")
            st.image(str(comp_path), caption="Perbandingan metric model di test set", width=700)

        # Feature importance
        fi_path = ASSETS_DIR / "feature_importance.png"
        if fi_path.exists():
            st.subheader("Feature Importance")
            st.image(str(fi_path), caption=f"Feature Importance — {metrics['best_model']}", width=600)
    else:
        st.info("Metrics belum tersedia. Jalankan `python -m src.train` terlebih dahulu.")

# ---------------------------------------------------------------------
# 4. TENTANG MODEL
# ---------------------------------------------------------------------

elif page == "ℹ️ Tentang Model":
    st.title("Tentang Model")

    metrics = load_metrics()
    if metrics is None:
        st.error("Metrics belum tersedia.")
        st.stop()

    st.markdown(f"""
    ### Model yang Digunakan

    Model final yang dipilih adalah **{metrics['best_model']}** karena memberikan
    **ROC-AUC tertinggi** pada test set. ROC-AUC dipilih sebagai metric utama
    karena dataset bersifat *imbalanced* (~85% risiko rendah vs ~15% risiko tinggi),
    sehingga accuracy saja tidak cukup representatif.

    ### Preprocessing

    - **Fitur numerik** (usia, rokok/hari, kolesterol, tekanan darah, BMI, detak jantung, gula darah):
      imputasi median + StandardScaler.
    - **Fitur kategorikal** (jenis kelamin, pendidikan, obat BP, riwayat stroke/hipertensi/diabetes, status merokok):
      imputasi modus + OneHotEncoder (`drop='if_binary'`).

    Preprocessing di-build menggunakan `ColumnTransformer` di dalam `Pipeline` sklearn
    sehingga hanya di-*fit* pada data latih dan konsisten saat inference.

    ### Evaluation Metrics (Test Set)

    | Metric | Nilai |
    |--------|-------|
    | ROC-AUC | {metrics['comparison'][metrics['best_model']]['roc_auc']:.4f} |
    | Accuracy | {metrics['comparison'][metrics['best_model']]['accuracy']:.4f} |
    | Precision | {metrics['comparison'][metrics['best_model']]['precision']:.4f} |
    | Recall | {metrics['comparison'][metrics['best_model']]['recall']:.4f} |
    | F1-score | {metrics['comparison'][metrics['best_model']]['f1']:.4f} |

    Threshold klasifikasi yang digunakan: **{metrics.get('optimal_threshold', 0.5)}**
    (dipilih berdasarkan maksimum F1 pada out-of-fold CV di data latih).

    ### Keterbatasan Model

    1. **Bukan diagnosis medis.** Hasil model adalah estimasi statistik dan tidak
       dapat menggantikan pemeriksaan dokter.
    2. **Populasi dataset terbatas.** Dataset Framingham didominasi warga
       Framingham, Massachusetts (mayoritas kulit putih), sehingga generalisasi
       ke populasi lain mungkin terbatas.
    3. **Class imbalance.** Model mungkin lebih baik memprediksi risiko rendah
       dibanding risiko tinggi karena proporsi kelas yang tidak seimbang.
    4. **Faktor yang tidak tercakup.** Genetik, riwayat keluarga, dan faktor
       gaya hidup lain tidak tersedia di dataset.
    5. **Bukan causal inference.** Feature importance menunjukkan asosiasi
       dengan prediksi, bukan hubungan sebab-akibat.
    """)
