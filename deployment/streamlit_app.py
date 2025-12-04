import streamlit as st
import pandas as pd
import pickle
import os

# MUST BE FIRST STREAMLIT CALL
st.set_page_config(
    page_title="Forecasting System",
    layout="wide"
)

# ------------------------------------------------
# Paths
# ------------------------------------------------
ARTIFACT_DIR = "deployment"

MODEL_PATH          = f"{ARTIFACT_DIR}/lightgbm_model.pkl"
FEATURES_PATH       = f"{ARTIFACT_DIR}/feature_columns.pkl"
BASELINE_METRICS    = f"{ARTIFACT_DIR}/baseline_metrics.pkl"
ARIMA_METRICS       = f"{ARTIFACT_DIR}/arima_metrics.pkl"
SARIMA_METRICS      = f"{ARTIFACT_DIR}/sarima_metrics.pkl"
LGBM_METRICS        = f"{ARTIFACT_DIR}/lightgbm_metrics.pkl"
SAMPLE_DATA_PATH    = f"{ARTIFACT_DIR}/store1_weekly_predictions.csv"

# ------------------------------------------------
# Utilities
# ------------------------------------------------
def verify_file(path: str):
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

# ------------------------------------------------
# Validate artifacts
# ------------------------------------------------
required = {
    "LightGBM Model": MODEL_PATH,
    "Feature Columns": FEATURES_PATH,
    "Sample Data": SAMPLE_DATA_PATH,
    "Baseline Metrics": BASELINE_METRICS,
    "ARIMA Metrics": ARIMA_METRICS,
    "SARIMA Metrics": SARIMA_METRICS,
    "LightGBM Metrics": LGBM_METRICS,
}

missing_any = False

for name, path in required.items():
    if not verify_file(path):
        missing(name)
        missing_any = True

if missing_any:
    st.warning("⚠️ Required files missing. Please upload all artifacts.")
    st.stop()

# ------------------------------------------------
# Load Everything
# ------------------------------------------------
model           = load_pickle(MODEL_PATH)
features        = load_pickle(FEATURES_PATH)
baseline        = load_pickle(BASELINE_METRICS)
arima           = load_pickle(ARIMA_METRICS)
sarima          = load_pickle(SARIMA_METRICS)
lgbm_metrics    = load_pickle(LGBM_METRICS)
sample_df       = load_csv(SAMPLE_DATA_PATH)

if any(x is None for x in [model, features, baseline, arima, sarima, lgbm_metrics, sample_df]):
    st.error("❌ One or more artifacts could not be loaded. Check pickle versions and file paths.")
    st.stop()

# ------------------------------------------------
# UI
# ------------------------------------------------
st.title("📈 Demand Planning & Forecasting System")
st.markdown("### Machine Learning Driven Forecasting for Retail & FMCG")

st.divider()

# ------------------------------------------------
# Prediction
# ------------------------------------------------
st.header("🏬 Store Sales Prediction using LightGBM")

st.write("Model loaded successfully. Preview of sample input data:")
st.dataframe(sample_df.head(), use_container_width=True)

if st.button("Run Forecast Now"):
    try:
        X = sample_df[features]
        preds = model.predict(X)
        sample_df["Forecast"] = preds

        st.success("Prediction Completed Successfully!")
        st.dataframe(sample_df.head(10), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction Failed: {str(e)}")
