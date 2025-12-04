import os
import pickle
import pandas as pd
import streamlit as st

# ============================================================
# Resolve deployment folder path
# ============================================================
DEPLOY_DIR = os.path.dirname(__file__)

def file_path(name: str) -> str:
    return os.path.join(DEPLOY_DIR, name)


# ============================================================
# Cached Loaders
# ============================================================
@st.cache_data
def load_model():
    path = file_path("lightgbm_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_features():
    path = file_path("feature_columns.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found at: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_metrics():
    metrics = {}
    files = {
        "LightGBM": "lightgbm_metrics.pkl",
        "SARIMA": "sarima_metrics.pkl",
        "ARIMA": "arima_metrics.pkl",
        "Baseline": "baseline_metrics.pkl"
    }
    for model_name, filename in files.items():
        path = file_path(filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                metrics[model_name] = pickle.load(f)
        else:
            metrics[model_name] = {"Error": f"{filename} file missing"}
    return metrics


@st.cache_data
def load_samples():
    samples = {}
    files = {
        "LightGBM Test Forecast": "lightgbm_test_forecast.csv",
        "LightGBM Validation Predictions": "lightgbm_val_predictions.csv",
        "Baseline Weekly Predictions": "baseline_weekly_predictions.csv",
        "Store1 Weekly Predictions": "store1_weekly_predictions.csv"
    }
    for label, filename in files.items():
        path = file_path(filename)
        if os.path.exists(path):
            samples[label] = pd.read_csv(path)
        else:
            samples[label] = pd.DataFrame({"Error": [f"{filename} missing"]})
    return samples


# ============================================================
# Page UI
# ============================================================
st.set_page_config(
    page_title="Demand Planning & Forecasting System",
    layout="wide",
)

st.title("📊 Demand Planning & Forecasting System")
st.markdown(
    "#### Machine Learning–Driven Forecasting for Retail & FMCG"
)

# ============================================================
# Debug (Temporary)
# ============================================================
with st.expander("🛠 DEBUG: File Listing (Temporary)"):
    try:
        st.write("Deployment directory:", DEPLOY_DIR)
        st.write("Files in deployment folder:", os.listdir(DEPLOY_DIR))
    except Exception as e:
        st.error(f"Error listing files: {e}")


# ============================================================
# Load all critical data
# ============================================================

try:
    model = load_model()
    features = load_features()
    metrics = load_metrics()
    samples = load_samples()
except Exception as e:
    st.error(f"⚠ Critical Error Loading Artifacts:\n\n{e}")
    st.stop()


# ============================================================
# UI Navigation Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(
    ["📈 Metrics Dashboard", "🧪 Sample Predictions", "🔍 Model Inputs"]
)


# ============================================================
# Metrics Tab
# ============================================================
with tab1:
    st.subheader("📈 Model Performance Comparison")

    for model_name, result in metrics.items():
        st.markdown(f"### {model_name}")

        if isinstance(result, dict):
            df = pd.DataFrame.from_dict(result, orient="index", columns=["Value"])
            st.table(df)
        else:
            st.warning(f"Metrics format issue for {model_name}")

        st.divider()


# ============================================================
# Predictions Tab
# ============================================================
with tab2:
    st.subheader("🧪 Prediction Samples")

    for title, df in samples.items():
        st.markdown(f"### {title}")
        st.dataframe(df.head(20), use_container_width=True)
        st.divider()


# ============================================================
# Features Tab
# ============================================================
with tab3:
    st.subheader("🔍 Model Feature Set")

    st.markdown("**Loaded Feature Columns Used by LightGBM Model:**")
    st.write(features)

    st.info(
        """
        These are the engineered features used to train the LightGBM model.  
        Ensure all incoming data during inference contains these fields.
        """
    )
