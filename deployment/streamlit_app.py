import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
DEPLOY_PATH = "deployment"


# -----------------------------
# HELPER: Validate File Paths
# -----------------------------
def assert_file(path):
    if not os.path.exists(path):
        st.error(f"❌ Missing required file: `{path}`")
        st.stop()


# -----------------------------
# HELPER: Load Pickle Files
# -----------------------------
@st.cache_resource
def load_pickle(filename):
    path = os.path.join(DEPLOY_PATH, filename)
    assert_file(path)
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------
# HELPER: Load CSV Files
# -----------------------------
@st.cache_data
def load_csv(filename):
    path = os.path.join(DEPLOY_PATH, filename)
    assert_file(path)
    return pd.read_csv(path)


# -----------------------------
# LOAD ALL ARTIFACTS
# -----------------------------
model = load_pickle("lightgbm_model.pkl")
metrics_lgb = load_pickle("lightgbm_metrics.pkl")
metrics_baseline = load_pickle("baseline_metrics.pkl")
metrics_arima = load_pickle("arima_metrics.pkl")
metrics_sarima = load_pickle("sarima_metrics.pkl")

feature_cols = load_pickle("feature_columns.pkl")

test_predictions = load_csv("lightgbm_test_forecast.csv")
baseline_results = load_csv("baseline_weekly_predictions.csv")
store1_results = load_csv("store1_weekly_predictions.csv")


# -----------------------------
# PAGE HEADER
# -----------------------------
st.set_page_config(page_title="Demand Forecasting System", layout="wide")

st.title("📊 Demand Planning & Forecasting System")
st.write("Machine Learning Driven Forecasting for Retail & FMCG")


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Choose a view:",
    ("📈 LightGBM Predictions", "🏪 Store Forecast Visualization", "📀 Model Performance Metrics"),
)


# -----------------------------
# PAGE 1: LightGBM TEST SET RESULTS
# -----------------------------
if page == "📈 LightGBM Predictions":
    st.subheader("LightGBM Test Set Forecast Results")
    st.dataframe(test_predictions.head(20))

    st.download_button(
        label="⬇ Download Full Forecast",
        data=test_predictions.to_csv(index=False),
        file_name="lightgbm_test_forecast.csv",
        mime="text/csv"
    )

    st.success("✔ Forecast loaded successfully.")


# -----------------------------
# PAGE 2: STORE VISUALIZATION
# -----------------------------
elif page == "🏪 Store Forecast Visualization":
    st.subheader("Store-Level Forecast Visualization")

    stores_available = sorted(test_predictions["Id"].unique())
    selected_store = st.selectbox("Select Store ID:", stores_available)

    df = store1_results.copy()
    df = df[df["Store"] == selected_store].copy()

    if df.empty:
        st.warning(f"No prediction data found for Store {selected_store}.")
    else:
        df["Date"] = pd.to_datetime(df["Date"])

        fig = plt.figure(figsize=(12, 5))
        plt.plot(df["Date"], df["LGB_Pred"], label=f"LightGBM Store {selected_store}", linewidth=2)
        plt.title(f"Store {selected_store} - LightGBM Forecast")
        plt.xlabel("Date")
        plt.ylabel("Predicted Sales")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        st.pyplot(fig)


# -----------------------------
# PAGE 3: MODEL METRICS
# -----------------------------
elif page == "📀 Model Performance Metrics":
    st.subheader("Model Evaluation Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚀 LightGBM Metrics")
        st.json(metrics_lgb)

        st.markdown("### 🧪 ARIMA Metrics")
        st.json(metrics_arima)

    with col2:
        st.markdown("### 📉 Baseline Metrics")
        st.json(metrics_baseline)

        st.markdown("### ⏳ SARIMA Metrics")
        st.json(metrics_sarima)

    st.info("These metrics were computed during model development.")


# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("📌 Built for Demand Planning & Forecasting Capstone Project")
st.markdown("🧠 Powered by Machine Learning")
