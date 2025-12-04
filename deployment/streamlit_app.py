import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------------------------
# Paths (Important!)
# -----------------------------------------
ARTIFACT_DIR = "deployment"

MODEL_PATH = f"{ARTIFACT_DIR}/lightgbm_model.pkl"
FEATURES_PATH = f"{ARTIFACT_DIR}/feature_columns.pkl"
BASELINE_METRICS_PATH = f"{ARTIFACT_DIR}/baseline_metrics.pkl"
ARIMA_METRICS_PATH = f"{ARTIFACT_DIR}/arima_metrics.pkl"
SARIMA_METRICS_PATH = f"{ARTIFACT_DIR}/sarima_metrics.pkl"
LGBM_METRICS_PATH = f"{ARTIFACT_DIR}/lightgbm_metrics.pkl"
SAMPLE_DATA_PATH = f"{ARTIFACT_DIR}/store1_weekly_predictions.csv"

# -----------------------------------------
# Utilities
# -----------------------------------------
def verify_file(path:str):
    """Check if a file exists."""
    return os.path.exists(path)

def missing(msg):
    st.error(f"❌ Missing file: `{msg}`")

# -----------------------------------------
# Loaders (cached)
# -----------------------------------------
@st.cache_data
def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
        return None

@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except:
        return None

# -----------------------------------------
# Load Everything
# -----------------------------------------
model = load_pickle(MODEL_PATH)
features = load_pickle(FEATURES_PATH)
baseline_metrics = load_pickle(BASELINE_METRICS_PATH)
arima_metrics = load_pickle(ARIMA_METRICS_PATH)
sarima_metrics = load_pickle(SARIMA_METRICS_PATH)
lightgbm_metrics = load_pickle(LGBM_METRICS_PATH)
sample_df = load_csv(SAMPLE_DATA_PATH)

# -----------------------------------------------------
# UI
# -----------------------------------------------------
st.set_page_config(page_title="Forecasting System", layout="wide")

st.title("📊 Demand Planning & Forecasting System")
st.markdown("### Machine Learning Driven Forecasting for Retail & FMCG")

# Check missing artifacts
required = {
    "Model File": MODEL_PATH,
    "Feature Columns": FEATURES_PATH,
    "Store Predictions Sample": SAMPLE_DATA_PATH,
}

missing_any = False
for key, path in required.items():
    if not verify_file(path):
        missing(path)
        missing_any = True

if missing_any:
    st.warning("⚠️ Model or required artifacts missing. Please verify uploaded files.")
    st.stop()

# -----------------------------------------------------
# SECTION 1 — Prediction
# -----------------------------------------------------
st.header("🛒 Store Sales Prediction using LightGBM")

st.write("Model loaded successfully. Preview sample input data:")
st.dataframe(sample_df.head())

# Predict Button
if st.button("Run Forecast Now"):
    try:
        X = sample_df[features]
        preds = model.predict(X)
        sample_df["Forecast"] = preds
        st.success("Prediction Completed Successfully!")
        st.dataframe(sample_df[["Forecast"]].head(10))
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -----------------------------------------------------
# SECTION 2 — Full Comparison
# -----------------------------------------------------
st.header("📊 Full Model Metrics Comparison")

cols = st.columns(2)

# ------------ Baseline
with cols[0]:
    st.subheader("📌 Baseline (Naive Model) Metrics")
    if baseline_metrics:
        st.json(baseline_metrics)
    else:
        missing(BASELINE_METRICS_PATH)

# ------------ ARIMA
with cols[1]:
    st.subheader("🔷 ARIMA Metrics")
    if arima_metrics:
        st.json(arima_metrics)
    else:
        missing(ARIMA_METRICS_PATH)

# ------------ SARIMA
with cols[0]:
    st.subheader("🔵 SARIMA Metrics")
    if sarima_metrics:
        st.json(sarima_metrics)
    else:
        missing(SARIMA_METRICS_PATH)

# ------------ LightGBM
with cols[1]:
    st.subheader("⚡ LightGBM Metrics")
    if lightgbm_metrics:
        st.json(lightgbm_metrics)
    else:
        missing(LGBM_METRICS_PATH)

# -----------------------------------------------------
# SECTION 3 — Model Selection
# -----------------------------------------------------
st.header("🧠 Best Performing Model")

st.success(
    "Based on evaluation metrics, **LightGBM** achieved the strongest forecasting performance."
)
