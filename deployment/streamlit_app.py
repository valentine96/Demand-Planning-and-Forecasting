import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------------------------
# MUST BE FIRST STREAMLIT CALL
# -----------------------------------------
st.set_page_config(
    page_title="Forecasting System",
    layout="wide"
)

# -----------------------------------------
# Paths
# -----------------------------------------
ARTIFACT_DIR = "deployment"

MODEL_PATH = f"{ARTIFACT_DIR}/lightgbm_model.pkl"
FEATURES_PATH = f"{ARTIFACT_DIR}/feature_columns.pkl"
BASELINE_METRICS_PATH = f"{ARTIFACT_DIR}/baseline_metrics.pkl"
ARIMA_METRICS_PATH = f"{ARTIFACT_DIR}/arima_metrics.pkl"
SARIMA_METRICS_PATH = f"{ARTIFACT_DIR}/sarima_metrics.pkl"
LGBM_METRICS_PATH = f"{ARTIFACT_DIR}/lightgbm_metrics.pkl"

# IMPORTANT CSV for inference
SAMPLE_DATA_PATH = f"{ARTIFACT_DIR}/deployment_full_features.csv"


# -----------------------------------------
# Utility Functions
# -----------------------------------------
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


# -----------------------------------------
# Validate Required Files
# -----------------------------------------
required = {
    "Model File": MODEL_PATH,
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
    st.warning("⚠ Model or required artifacts missing. Please verify uploaded files.")
    st.stop()


# -----------------------------------------
# Load Everything
# -----------------------------------------
model = load_pickle(MODEL_PATH)
features = load_pickle(FEATURES_PATH)
baseline_metrics = load_pickle(BASELINE_METRICS_PATH)
arima_metrics = load_pickle(ARIMA_METRICS_PATH)
sarima_metrics = load_pickle(SARIMA_METRICS_PATH)
lgbm_metrics = load_pickle(LGBM_METRICS_PATH)
sample_df = load_csv(SAMPLE_DATA_PATH)


# -----------------------------------------
# Validate load success
# -----------------------------------------
if any(x is None for x in [
    model, features, baseline_metrics,
    arima_metrics, sarima_metrics, lgbm_metrics,
    sample_df
]):
    st.error("❌ One or more artifacts failed to load. Check file versions.")
    st.stop()


# -----------------------------------------
# Validate Feature Alignment
# -----------------------------------------
missing_features = [c for c in features if c not in sample_df.columns]

if missing_features:
    st.error(f"""
    ❌ Some feature columns expected by the model are missing in the input data.

    Missing columns: {missing_features}

    Ensure that the CSV used for deployment contains the full engineered features.
    """)
    st.stop()


st.success("✔ All required files found and validated!")
st.write(f"📌 Loaded model expects **{len(features)} features**.")
st.write(f"📌 Sample data contains **{len(sample_df.columns)} columns**.")


# -----------------------------------------
# UI HEADER
# -----------------------------------------
st.title("📈 Demand Planning & Forecasting System")
st.markdown("### Machine Learning Driven Forecasting for Retail & FMCG")

st.divider()


# -----------------------------------------
# SECTION 1 – Sample Preview
# -----------------------------------------
st.header("🔍 Preview Input Data Used For Prediction")

default_rows = 5
max_rows = min(50, len(sample_df))

rows_to_preview = st.slider(
    "Rows to preview",
    min_value=1,
    max_value=max_rows,
    value=default_rows
)
st.dataframe(sample_df.head(rows_to_preview), use_container_width=True)


# -----------------------------------------
# Store Selector (Optional)
# -----------------------------------------
if "Store" in sample_df.columns:
    store_ids = sorted(sample_df["Store"].unique())
    selected_store = st.selectbox("Select store to preview forecast", store_ids)

    store_df = sample_df[sample_df["Store"] == selected_store]
    st.write(f"Preview for store: {selected_store}")
    st.dataframe(store_df.head(rows_to_preview))
else:
    store_df = sample_df  # fallback


st.divider()


# -----------------------------------------
# SECTION 2 – Forecast
# -----------------------------------------
st.header("📦 Store Sales Prediction using LightGBM")

st.write("Model loaded successfully. Preview sample input data:")
st.dataframe(sample_df.head(), use_container_width=True)

if st.button("🚀 Run Forecast Now"):
    try:
        X = sample_df[features]
        preds = model.predict(X)
        sample_df["Forecast"] = preds

        st.success("🎉 Prediction Completed Successfully!")
        st.dataframe(sample_df.head(20), use_container_width=True)

        # Download link
        csv_download = sample_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Forecast Results",
            csv_download,
            "predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")


st.divider()


# -----------------------------------------
# SECTION 3 – Model Metrics Comparison
# -----------------------------------------
st.header("📊 Full Model Metrics Comparison")

left, right = st.columns(2)

with left:
    st.subheader("📌 Baseline (Naive Model) Metrics")
    if baseline_metrics:
        st.json(baseline_metrics)
    else:
        st.warning("No baseline metrics found.")

    st.subheader("📌 SARIMA Metrics")
    if sarima_metrics:
        st.json(sarima_metrics)
    else:
        st.warning("No SARIMA metrics found.")


with right:
    st.subheader("📌 ARIMA Metrics")
    if arima_metrics:
        st.json(arima_metrics)
    else:
        st.warning("No ARIMA metrics found.")

    st.subheader("📌 LightGBM Metrics")
    if lgbm_metrics:
        st.json(lgbm_metrics)
    else:
        st.warning("No LightGBM metrics found.")


st.success("🎯 LightGBM achieved the strongest forecasting performance based on evaluation metrics.")
