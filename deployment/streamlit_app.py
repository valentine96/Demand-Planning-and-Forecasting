import streamlit as st
import pandas as pd
import pickle
import base64

# -------------------------------------------------------
#  CONFIG
# -------------------------------------------------------
DEPLOY_DIR = "."   # Since app runs inside deployment folder on Streamlit Cloud
st.set_page_config(page_title="Demand Forecasting System", layout="wide")

# -------------------------------------------------------
#  LOADER HELPERS (CACHED)
# -------------------------------------------------------
@st.cache_data
def load_pickle(file_name):
    file_path = f"{DEPLOY_DIR}/{file_name}"
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Missing file: `{file_name}`")
        return None

# Model
@st.cache_data
def load_model():
    return load_pickle("lightgbm_model.pkl")

# Feature Columns
@st.cache_data
def load_features():
    return load_pickle("feature_columns.pkl")

# Metrics
@st.cache_data
def load_baseline_metrics():
    return load_pickle("baseline_metrics.pkl")

@st.cache_data
def load_arima_metrics():
    return load_pickle("arima_metrics.pkl")

@st.cache_data
def load_sarima_metrics():
    return load_pickle("sarima_metrics.pkl")

@st.cache_data
def load_lgbm_metrics():
    return load_pickle("lightgbm_metrics.pkl")

# Predictions Sample
@st.cache_data
def load_store_sample():
    return load_pickle("store1_weekly_predictions.csv")


# -------------------------------------------------------
#   MAIN APP UI
# -------------------------------------------------------

st.title("📈 Demand Planning & Forecasting System")
st.markdown("#### Machine Learning Driven Forecasting for Retail & FMCG")

menu = st.sidebar.radio(
    "Navigation",
    ["Forecast", "Model & Metrics Overview"]
)

# -------------------------------------------------------
# PAGE 1 - FORECAST VIEW
# -------------------------------------------------------
if menu == "Forecast":
    st.subheader("🛍 Store Sales Prediction using LightGBM")

    model = load_model()
    feature_cols = load_features()
    sample_df = load_store_sample()

    if model is None or feature_cols is None or sample_df is None:
        st.warning("⚠ Model or required artifacts missing. Please verify uploaded files.")
    else:
        st.success("Model loaded successfully! 👌")

        stores = sorted(sample_df["Store"].unique().tolist())
        store_id = st.selectbox("Select Store:", stores)

        store_data = (
            sample_df[sample_df["Store"] == store_id]
            [["Date", "LGB_Pred"]]
        )

        store_data["Date"] = pd.to_datetime(store_data["Date"])

        st.line_chart(
            store_data.set_index("Date")["LGB_Pred"],
            height=400
        )

        st.write(f"📌 Showing LightGBM forecast for Store **{store_id}**")

# -------------------------------------------------------
# PAGE 2 - METRICS OVERVIEW
# -------------------------------------------------------
elif menu == "Model & Metrics Overview":
    st.subheader("📊 Full Model Metrics Comparison")

    col1, col2 = st.columns(2)

    # Baseline Metrics
    with col1:
        st.markdown("### 📌 Baseline (Naive Model) Metrics")
        baseline = load_baseline_metrics()
        st.write(baseline if baseline is not None else "No metrics found.")

    # ARIMA
    with col2:
        st.markdown("### 🔁 ARIMA Metrics")
        arima = load_arima_metrics()
        st.write(arima if arima is not None else "No metrics found.")

    col3, col4 = st.columns(2)

    # SARIMA
    with col3:
        st.markdown("### 🌀 SARIMA Metrics")
        sarima = load_sarima_metrics()
        st.write(sarima if sarima is not None else "No metrics found.")

    # LightGBM
    with col4:
        st.markdown("### ⚡ LightGBM Metrics")
        lgbm = load_lgbm_metrics()
        st.write(lgbm if lgbm is not None else "No metrics found.")

    st.divider()
    st.markdown("### 🧠 Best Performing Model")
    st.success("Based on evaluation metrics, **LightGBM** achieved the strongest forecasting performance.")
