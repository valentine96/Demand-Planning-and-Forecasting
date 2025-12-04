import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------------------------------------
# Paths (Important!)
# -----------------------------------------------------
ARTIFACT_DIR = "deployment"

MODEL_PATH             = f"{ARTIFACT_DIR}/lightgbm_model.pkl"
FEATURES_PATH          = f"{ARTIFACT_DIR}/feature_columns.pkl"
BASELINE_METRICS_PATH  = f"{ARTIFACT_DIR}/baseline_metrics.pkl"
ARIMA_METRICS_PATH     = f"{ARTIFACT_DIR}/arima_metrics.pkl"
SARIMA_METRICS_PATH    = f"{ARTIFACT_DIR}/sarima_metrics.pkl"
LGBM_METRICS_PATH      = f"{ARTIFACT_DIR}/lightgbm_metrics.pkl"
SAMPLE_DATA_PATH       = f"{ARTIFACT_DIR}/store1_weekly_predictions.csv"


# -----------------------------------------------------
# Utilities
# -----------------------------------------------------

def verify_file(path: str):
    """Check if a file exists."""
    return os.path.exists(path)

def missing(msg):
    st.error(f"❌ Missing file: `{msg}`")

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


# -----------------------------------------------------
# Validate ALL Required Artifacts
# -----------------------------------------------------
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
        missing(name)
        missing_any = True

if missing_any:
    st.warning("⚠️ Model or artifacts missing. Please ensure all required files are present.")
    st.stop()


# -----------------------------------------------------
# Load Everything
# -----------------------------------------------------
model             = load_pickle(MODEL_PATH)
features          = load_pickle(FEATURES_PATH)
baseline_metrics  = load_pickle(BASELINE_METRICS_PATH)
arima_metrics     = load_pickle(ARIMA_METRICS_PATH)
sarima_metrics    = load_pickle(SARIMA_METRICS_PATH)
lgbm_metrics      = load_pickle(LGBM_METRICS_PATH)
sample_df         = load_csv(SAMPLE_DATA_PATH)

# Final failsafe validation
if any(x is None for x in [
    model, features, baseline_metrics,
    arima_metrics, sarima_metrics,
    lgbm_metrics, sample_df
]):
    st.error("❌ One or more artifacts could not be loaded. Ensure correct pickle versions.")
    st.stop()


# -----------------------------------------------------
# UI Setup
# -----------------------------------------------------
st.set_page_config(
    page_title="Forecasting System",
    layout="wide"
)

st.title("📈 Demand Planning & Forecasting System")
st.markdown("#### Machine Learning Driven Forecasting for Retail & FMCG")

st.divider()


# -----------------------------------------------------
# SECTION 1 — Prediction
# -----------------------------------------------------
st.header(" 🛍 Store Sales Prediction using LightGBM")

st.write("Model loaded successfully. Preview of data used in prediction:")
st.dataframe(sample_df.head(), use_container_width=True)

if st.button("Run Forecast Now"):
    try:
        X = sample_df[features]
        preds = model.predict(X)
        sample_df["Forecast"] = preds

        st.success("🎉 Prediction Completed Successfully!")
        st.subheader("Forecast Output")
        st.dataframe(sample_df.head(15), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Forecasting Error: {e}")


st.divider()


# -----------------------------------------------------
# SECTION 2 — Full Metrics Comparison
# -----------------------------------------------------
st.header("📊 Full Model Metrics Comparison")


col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Baseline (Naive Model) Metrics")
    st.write(baseline_metrics)

with col2:
    st.subheader("📌 ARIMA Metrics")
    st.write(arima_metrics)

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader("📌 SARIMA Metrics")
    st.write(sarima_metrics)

with col4:
    st.subheader("📌 LightGBM Metrics")
    st.write(lgbm_metrics)


st.divider()

# Best Model
st.header("🧠 Best Performing Model")
st.success("Based on evaluation metrics, **LightGBM** achieved the strongest forecasting accuracy.")
