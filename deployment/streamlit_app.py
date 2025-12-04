import streamlit as st
import pandas as pd
import pickle
import os

# ------------------------------------------------------------------------------
# MUST BE FIRST STREAMLIT CALL
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Forecasting System",
    layout="wide"
)

# ------------------------------------------------------------------------------
# Paths (Important!)
# ------------------------------------------------------------------------------
ARTIFACT_DIR = "deployment"

MODEL_PATH = f"{ARTIFACT_DIR}/lightgbm_model.pkl"
FEATURES_PATH = f"{ARTIFACT_DIR}/feature_columns.pkl"
BASELINE_METRICS_PATH = f"{ARTIFACT_DIR}/baseline_metrics.pkl"
ARIMA_METRICS_PATH = f"{ARTIFACT_DIR}/arima_metrics.pkl"
SARIMA_METRICS_PATH = f"{ARTIFACT_DIR}/sarima_metrics.pkl"
LGBM_METRICS_PATH = f"{ARTIFACT_DIR}/lightgbm_metrics.pkl"
SAMPLE_DATA_PATH = f"{ARTIFACT_DIR}/deployment_full_features.csv"

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def verify_file(path: str):
    return os.path.exists(path)

def missing(msg):
    st.error(f"❌ Missing file: `{msg}`")

@st.cache_data
def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

# ------------------------------------------------------------------------------
# Validate Required Artifacts
# ------------------------------------------------------------------------------
required = {
    "LightGBM Model": MODEL_PATH,
    "Feature Columns": FEATURES_PATH,
    "Sample Data": SAMPLE_DATA_PATH,
    "Baseline Metrics": BASELINE_METRICS_PATH,
    "ARIMA Metrics": ARIMA_METRICS_PATH,
    "SARIMA Metrics": SARIMA_METRICS_PATH,
    "LightGBM Metrics": LGBM_METRICS_PATH,
}

missing_any = False

for name, path in required.items():
    if not verify_file(path):
        missing(path)
        missing_any = True

if missing_any:
    st.warning("⚠️ Model or artifacts missing. Please ensure all required files are present.")
    st.stop()

# ------------------------------------------------------------------------------
# Load Everything
# ------------------------------------------------------------------------------
model = load_pickle(MODEL_PATH)
features = load_pickle(FEATURES_PATH)
baseline_metrics = load_pickle(BASELINE_METRICS_PATH)
arima_metrics = load_pickle(ARIMA_METRICS_PATH)
sarima_metrics = load_pickle(SARIMA_METRICS_PATH)
lgbm_metrics = load_pickle(LGBM_METRICS_PATH)
sample_df = load_csv(SAMPLE_DATA_PATH)

# Fail-safe
if any(x is None for x in [
    model, features, baseline_metrics, arima_metrics,
    sarima_metrics, lgbm_metrics, sample_df
]):
    st.error("❌ One or more artifacts could not be loaded. Ensure correct pickle versions.")
    st.stop()

# ------------------------------------------------------------------------------
# UI Header
# ------------------------------------------------------------------------------
st.title("🧠 Demand Planning & Forecasting System")
st.markdown("### Machine Learning Driven Forecasting for Retail & FMCG")

st.success(f"📌 Loaded model expects **{features.shape[1]} features**.")
st.info(f"📄 Sample data contains **{sample_df.shape[1]} columns**.")

st.divider()

# ------------------------------------------------------------------------------
# FIX: Align categorical features with training data
# ------------------------------------------------------------------------------
st.subheader("🔧 Feature Alignment Check")

categorical_cols = ["StoreType", "Assortment", "PromoInterval", "StateHoliday"]

aligned_categories = []

for col in categorical_cols:
    if col in sample_df.columns and col in features.columns:
        # Convert to category
        sample_df[col] = sample_df[col].astype("category")
        # Align categories to those used during training
        sample_df[col] = sample_df[col].cat.set_categories(
            features[col].cat.categories
        )
        aligned_categories.append(col)

if aligned_categories:
    st.success(f"✔️ Categorical features aligned: {', '.join(aligned_categories)}")
else:
    st.warning("⚠️ No categorical alignment needed or columns not found.")

st.divider()

# ------------------------------------------------------------------------------
# DATA PREVIEW
# ------------------------------------------------------------------------------
st.header("🔍 Preview Input Data Used For Prediction")

rows = st.slider("Rows to preview", 1, 50, 5)

st.dataframe(sample_df.head(rows), use_container_width=True)

st.divider()

# ------------------------------------------------------------------------------
# SECTION 1 — PREDICTION
# ------------------------------------------------------------------------------
st.header("📦 Store Sales Prediction using LightGBM")

st.write("Model loaded successfully. Preview sample input data:")
st.dataframe(sample_df.head(), use_container_width=True)

if st.button("🚀 Run Forecast Now"):
    try:
        X = sample_df[features.columns]
        preds = model.predict(X)
        sample_df["Forecast"] = preds

        st.success("🎉 Prediction Completed Successfully!")
        st.dataframe(sample_df[["Date","Store","Forecast"]].head(10), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

st.divider()

# ------------------------------------------------------------------------------
# MODEL METRICS
# ------------------------------------------------------------------------------
st.header("📊 Full Model Metrics Comparison")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Baseline (Naive Model) Metrics")
    st.json(baseline_metrics)

with col2:
    st.subheader("📌 ARIMA Metrics")
    st.json(arima_metrics)

col3, col4 = st.columns(2)

with col3:
    st.subheader("📌 SARIMA Metrics")
    st.json(sarima_metrics)

with col4:
    st.subheader("📌 LightGBM Metrics")
    st.json(lgbm_metrics)

st.success("🎯 LightGBM achieved the strongest forecasting performance based on evaluation metrics.")
